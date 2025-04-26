package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Artists_getArtist_3_2_Test {

    @Test
    public void testGetArtist() {
        Artists artists = mock(Artists.class);
        when(artists.getArtist(0)).thenReturn("Artist1");
        when(artists.getArtist(1)).thenReturn("Artist2");
        assertEquals("Artist1", artists.getArtist(0));
        assertEquals("Artist2", artists.getArtist(1));
        assertEquals(null, artists.getArtist(2));
    }
}
