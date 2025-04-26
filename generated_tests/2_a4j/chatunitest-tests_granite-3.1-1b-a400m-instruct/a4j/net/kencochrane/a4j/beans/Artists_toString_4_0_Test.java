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

    private Artists artists;

    @BeforeEach
    void setUp() {
        artists = new Artists();
        artists.setArtist(new String[] { "Alice", "Bob", "Charlie" });
        artists.setArtist(new String[] { "Dave", "Eve", "Frank" });
        artists.setArtist(new String[] { "Grace", "Harry", "Ivy" });
    }

    @Test
    void testToString() {
        String expectedOutput = "# of Lists = 3\nartist - Alice\nartist - Bob\nartist - Charlie\n# of Lists = 3\n";
        String actualOutput = artists.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
