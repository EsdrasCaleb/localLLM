package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Artists_getArtist_3_0_Test {

    @Mock
    Artists artists;

    @BeforeEach
    public void setup() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testGetArtist() {
        // Given
        String[] artistNames = { "Artist1", "Artist2", "Artist3" };
        when(artists.getArtist()).thenReturn(artistNames);
        // When
        String artist = artists.getArtist(0);
        // Then
        assertEquals("Artist1", artist);
    }
}
