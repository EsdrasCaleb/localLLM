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

public class Artists_toString_4_0_Test {

    private Artists artists;

    @BeforeEach
    public void setUp() {
        artists = new Artists();
    }

    @Test
    public void testToStringWithNullArtists() throws NoSuchFieldException, IllegalAccessException {
        // Set artists field to null using reflection
        Field artistsField = Artists.class.getDeclaredField("artists");
        artistsField.setAccessible(true);
        artistsField.set(artists, null);
        assertEquals("artists is null or size 0 \n", artists.toString());
    }

    @Test
    public void testToStringWithEmptyArtists() {
        // Initialize artists as an empty ArrayList
        artists.setArtist(new String[0]);
        assertEquals("artists is null or size 0 \n", artists.toString());
    }

    @Test
    public void testToStringWithOneArtist() {
        // Initialize artists with one artist
        artists.setArtist(new String[] { "Picasso" });
        assertEquals("# of Lists = 1\nartist - Picasso\n", artists.toString());
    }

    @Test
    public void testToStringWithMultipleArtists() {
        // Initialize artists with multiple artists
        artists.setArtist(new String[] { "Picasso", "Van Gogh", "Da Vinci" });
        assertEquals("# of Lists = 3\nartist - Picasso\nartist - Van Gogh\nartist - Da Vinci\n", artists.toString());
    }
}
