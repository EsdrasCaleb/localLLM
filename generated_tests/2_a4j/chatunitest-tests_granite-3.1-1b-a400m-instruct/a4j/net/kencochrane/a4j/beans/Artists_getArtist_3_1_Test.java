package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Artists_getArtist_3_1_Test {

    @Test
    void testGetArtist() {
        Artists artists = new Artists();
        artists.setArtist(new String[] { "Artist 1", "Artist 2" });
        assertEquals("Artist 1", artists.getArtist(0));
        assertEquals("Artist 2", artists.getArtist(1));
        assertEquals("Artist 1", artists.getArtist(2));
        assertEquals("Artist 2", artists.getArtist(3));
        // Index out of range
        assertEquals("Artist 1", artists.getArtist(4));
        assertNull("Artist 3", artists.getArtist(5));
    }
}
