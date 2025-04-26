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

class Artists_toString_4_0_Test {

    @Test
    void testToStringEmptyList() {
        Artists artists = new Artists();
        String expected = "artists is null or size 0 \n";
        assertEquals(expected, artists.toString());
    }

    @Test
    void testToStringNullList() {
        Artists artists = new Artists();
        artists.artists = null;
        String expected = "artists is null or size 0 \n";
        assertEquals(expected, artists.toString());
    }

    @Test
    void testToStringNonEmptyList() {
        Artists artists = new Artists();
        artists.setArtist(new String[] { "artist1", "artist2", "artist3" });
        String expected = "# of Lists = 3\nartist - artist1\nartist - artist2\nartist - artist3\n";
        assertEquals(expected, artists.toString());
    }

    @Test
    void testToStringWithNullArtist() {
        Artists artists = new Artists();
        artists.setArtist(new String[] { "artist1", null, "artist3" });
        String expected = "# of Lists = 3\nartist - artist1\nartist - null\nartist - artist3\n";
        assertEquals(expected, artists.toString());
    }

    @Test
    void testToStringWithOneArtist() {
        Artists artists = new Artists();
        artists.setArtist(new String[] { "artist1" });
        String expected = "# of Lists = 1\nartist - artist1\n";
        assertEquals(expected, artists.toString());
    }
}
