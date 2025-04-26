package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class Artists_getArtist_3_0_Test {

    // Test class
    @Test
    public void testGetArtist() {
        Artists artists = new Artists();
        artists.setArtist(new String[] { "Artist 1", "Artist 2", "Artist 3" });
        assertEquals("Artist 1", artists.getArtist(0));
        assertEquals("Artist 2", artists.getArtist(1));
        assertEquals("Artist 3", artists.getArtist(2));
        assertEquals(null, artists.getArtist(3));
    }
}
