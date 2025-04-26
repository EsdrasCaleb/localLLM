package net.kencochrane.a4j.beans;

import java.io.Serializable;
import java.util.ArrayList;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class Platforms_toString_4_4_Test {

    @ExtendWith(MockitoExtension.class)
    public class TestPlatforms_toString_4_4 {

        @Test
        public void testToString() {
            // Arrange
            ArrayList<String> platforms = new ArrayList<>();
            when(platforms.size()).thenReturn(3);
            when(platforms.get(0)).thenReturn("Apple");
            when(platforms.get(1)).thenReturn("Banana");
            when(platforms.get(2)).thenReturn("Cherry");
            // Act
            String result = platforms.toString();
            // Assert
            assertEquals("Number of Platforms = 3\nPlatform - Apple\nPlatform - Banana\nPlatform - Cherry", result);
        }

        @Test
        public void testToString_Buggy_Liene() {
            // Arrange
            ArrayList<String> platforms = new ArrayList<>();
            when(platforms.size()).thenReturn(3);
            when(platforms.get(0)).thenReturn("Apple");
            when(platforms.get(1)).thenReturn("Banana");
            when(platforms.get(2)).thenReturn("Cherry");
            // Act
            String result = platforms.toString();
            // Assert
            assertEquals("Number of Platforms = 3\nPlatform - Apple\nPlatform - Banana\nPlatform - Cherry", result);
        }
    }
}
