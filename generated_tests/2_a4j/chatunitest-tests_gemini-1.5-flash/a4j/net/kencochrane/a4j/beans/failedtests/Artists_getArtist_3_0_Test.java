package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Artists_getArtist_3_0_Test {

    private Artists artists;

    @BeforeEach
    void setUp() {
        artists = new Artists();
    }

    @Test
    void testGetArtistWithinBounds() {
        String[] artistNames = { "Picasso", "Monet", "Van Gogh" };
        artists.setArtist(artistNames);
        assertEquals("Monet", artists.getArtist(1));
    }

    @Test
    void testGetArtistOutOfBounds() {
        String[] artistNames = { "Picasso", "Monet", "Van Gogh" };
        artists.setArtist(artistNames);
        assertNull(artists.getArtist(3));
    }

    @Test
    void testGetArtistEmptyList() {
        String[] artistNames = {};
        artists.setArtist(artistNames);
        assertNull(artists.getArtist(0));
    }

    @Test
    void testGetArtistNegativeIndex() {
        String[] artistNames = { "Picasso", "Monet", "Van Gogh" };
        artists.setArtist(artistNames);
        assertNull(artists.getArtist(-1));
    }

    @Test
    void testGetArtistListWithNull() {
        ArrayList<String> nullList = new ArrayList<>();
        nullList.add(null);
        try {
            Field artistsField = Artists.class.getDeclaredField("artists");
            artistsField.setAccessible(true);
            artistsField.set(artists, nullList);
            assertNull(artists.getArtist(0));
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Exception thrown: " + e.getMessage());
        }
    }
}
