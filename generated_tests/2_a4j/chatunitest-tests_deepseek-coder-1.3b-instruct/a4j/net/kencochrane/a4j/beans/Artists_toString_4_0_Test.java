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
    public void setup() {
        artists = new Artists();
    }

    @Test
    @DisplayName("Test with null artists list")
    public void testWithNullArtistsList() {
        artists = new Artists();
        String expected = "# of Lists = 0\nartists is null or size 0 \n";
        assertEquals(expected, artists.toString());
    }

    @Test
    @DisplayName("Test with empty artists list")
    public void testWithEmptyArtistsList() {
        artists.setArtist(new String[] {});
        String expected = "# of Lists = 0\nartists is null or size 0 \n";
        assertEquals(expected, artists.toString());
    }

    @Test
    @DisplayName("Test with a list of artists")
    public void testWithListOfArtists() {
        artists.setArtist(new String[] { "Artist1", "Artist2", "Artist3" });
        String expected = "# of Lists = 3\nartist - Artist1\nartist - Artist2\nartist - Artist3\n";
        assertEquals(expected, artists.toString());
    }
}
