package net.kencochrane.a4j.beans;

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
    public void testToString_WithArtists() {
        String[] artistNames = { "Artist1", "Artist2", "Artist3" };
        artists.setArtist(artistNames);
        String expectedOutput = "# of Lists = 3\n" + "artist - Artist1\n" + "artist - Artist2\n" + "artist - Artist3\n";
        assertEquals(expectedOutput, artists.toString());
    }

    @Test
    public void testToString_EmptyArtists() {
        artists.setArtist(new String[0]);
        String expectedOutput = "# of Lists = 0\n";
        assertEquals(expectedOutput, artists.toString());
    }

    @Test
    public void testToString_NullArtists() {
        // Using reflection to set artists to null
        try {
            java.lang.reflect.Field field = Artists.class.getDeclaredField("artists");
            field.setAccessible(true);
            field.set(artists, null);
        } catch (Exception e) {
            e.printStackTrace();
        }
        String expectedOutput = "artists is null or size 0 \n";
        assertEquals(expectedOutput, artists.toString());
    }
}
