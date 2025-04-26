package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Artists_getArtist_3_0_Test {

    private Artists artists;

    @BeforeEach
    public void setUp() {
        artists = mock(Artists.class);
    }

    @Test
    public void testGetArtistWithIndexInBounds() {
        String artistName = "John Doe";
        Mockito.when(artists.getArtist(0)).thenReturn(artistName);
        assertEquals(artistName, artists.getArtist(0));
    }

    @Test
    public void testGetArtistWithIndexOutOfBounds() {
        Mockito.when(artists.getArtist(10)).thenReturn(null);
        assertNull(artists.getArtist(10));
    }
}
