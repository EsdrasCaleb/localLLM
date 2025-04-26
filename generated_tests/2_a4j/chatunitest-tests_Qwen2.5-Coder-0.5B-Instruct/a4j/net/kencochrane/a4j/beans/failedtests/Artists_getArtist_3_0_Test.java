// Test method
package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class Artists_getArtist_3_0_Test {

    @Mock
    private Artists artistsMock;

    @Test
    public void testGetArtist() {
        // Arrange
        when(artistsMock.getArtist(0)).thenReturn("Artist 1");
        when(artistsMock.getArtist(2)).thenReturn("Artist 3");
        // Act
        String result = artistsMock.getArtist(1);
        // Assert
        assertEquals("Artist 1", result);
    }

    @ExtendWith(MockitoExtension.class)
    public static class TestArtists {

        @Test
        public void testGetArtist() {
            // Arrange
            Artists artistsMock = mock(Artists.class);
            when(artistsMock.getArtist(0)).thenReturn("Artist 1");
            when(artistsMock.getArtist(2)).thenReturn("Artist 3");
            // Act
            String result = artistsMock.getArtist(1);
            // Assert
            assertEquals("Artist 1", result);
        }
    }
}
