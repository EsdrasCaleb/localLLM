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

    @InjectMocks
    private Artists artists;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testToStringWithNonEmptyArtists() {
        String[] artistArray = { "Artist1", "Artist2", "Artist3" };
        artists.setArtist(artistArray);
        String expected = "# of Lists = 3\nartist - Artist1\nartist - Artist2\nartist - Artist3\n";
        String actual = artists.toString();
        assertEquals(expected, actual);
    }

    @Test
    void testToStringWithEmptyArtists() {
        String[] artistArray = {};
        artists.setArtist(artistArray);
        String expected = "artists is null or size 0 \n";
        String actual = artists.toString();
        assertEquals(expected, actual);
    }
}
