<script setup lang="ts">
import { VideoPlayer } from "@videojs-player/vue/dist/videojs-player.cjs.js";
const props = defineProps<{ src: string, poster: string }>()
</script>

<template>
    <VideoPlayer class="vjs-fluid vjs-16-9" :src="props.src" :poster="props.poster" :controls="true" />
</template>