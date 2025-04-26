package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Artists_toString_4_0_Test {

    @Test
    public void testToString() throws Exception {
        Artists artists = new Artists();
        // Test with null artists
        Field field = Artists.class.getDeclaredField("artists");
        field.setAccessible(true);
        field.set(artists, null);
        String expected = "artists is null or size 0 \n";
        assertEquals(expected, artists.toString());
        // Test with empty artists
        artists.setArtist(new String[0]);
        expected = "artists is null or size 0 \n";
        assertEquals(expected, artists.toString());
        // Test with non-empty artists
        artists.setArtist(new String[] { "artist1", "artist2", "artist3" });
        expected = "# of Lists = 3\nartist - artist1\nartist - artist2\nartist - artist3\n";
        assertEquals(expected, artists.toString());
    }
}
