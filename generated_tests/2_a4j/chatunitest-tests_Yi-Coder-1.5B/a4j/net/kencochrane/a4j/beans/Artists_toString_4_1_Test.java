package net.kencochrane.a4j.beans;

import java.util.concurrent.TimeUnit;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Artists_toString_4_1_Test {

    Artists artists;

    @BeforeEach
    void setup() {
        artists = new Artists();
    }

    @Test
    void testToString() {
        artists.setArtist(new String[] { "Artist1", "Artist2" });
        String expected = "# of Lists = 2\n" + "artist - Artist1\n" + "artist - Artist2\n";
        assertEquals(expected, artists.toString());
    }
}
