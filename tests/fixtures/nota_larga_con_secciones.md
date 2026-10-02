---
created: 2026-09-10
title: Guía de configuración del homelab
tags:
  - proyecto
  - tecnologia
  - homelab
---

# Guía de configuración del homelab

Este documento describe toda la configuración del homelab casero, incluyendo los servicios principales y las decisiones de arquitectura tomadas durante el proceso de montaje.

## Servicios principales

El homelab ejecuta varios servicios dockerizados: Nextcloud para almacenamiento de archivos, Gitea como servidor Git local, y Grafana junto con Prometheus para monitorización. Cada servicio corre en su propio contenedor con volúmenes persistentes montados en un disco SSD de 1TB.

## Configuración de red

### Router y DNS

El router principal tiene soporte de VLANs para separar el tráfico: una VLAN para dispositivos de confianza (ordenadores, móviles), otra para IoT (bombillas, sensores, aspiradora) y otra para el homelab. Esto asegura que un dispositivo IoT comprometido no pueda acceder al homelab ni a los dispositivos personales. El DNS interno usa un servidor DNS local corriendo en un mini-PC, que también sirve como servidor DHCP. El DNS bloquea publicidad y trackers a nivel de red, además de resolver nombres locales como gitea.home.arpa o nextcloud.home.arpa. La configuración del DNS está versionada en un repositorio Git para poder restaurarla fácilmente si el equipo falla.

### Certificados TLS

Para HTTPS interno usamos un certificado wildcard de Let's Encrypt para *.home.arpa, renovado automáticamente con certbot en modo DNS challenge. Así todos los servicios tienen HTTPS válido sin warnings del navegador. El certificado se distribuye a los servicios via un volumen Docker compartido que se actualiza cada 60 días. Traefik actúa como reverse proxy y se encarga de presentar el certificado correcto según el hostname del request. La configuración de Traefik incluye middleware para headers de seguridad, rate limiting, y redirección automática de HTTP a HTTPS.

### Firewall y acceso remoto

El acceso remoto al homelab se hace exclusivamente por VPN. No hay puertos abiertos al exterior excepto el de la VPN. Cada dispositivo que quiere acceder remotamente tiene sus propias credenciales. Las reglas del firewall en el router bloquean todo el tráfico entrante excepto la VPN, y el tráfico entre VLANs está restringido: IoT no puede hablar con homelab, pero homelab sí puede hablar con IoT (para monitorización con Prometheus).

## Backups

Los backups se hacen con restic hacia un bucket de Backblaze B2 (almacenamiento cloud barato). Restic encripta y deduplica automáticamente, así que el almacenamiento es eficiente. Se ejecuta diariamente a las 3AM via cron y envía una notificación por Telegram si falla.
