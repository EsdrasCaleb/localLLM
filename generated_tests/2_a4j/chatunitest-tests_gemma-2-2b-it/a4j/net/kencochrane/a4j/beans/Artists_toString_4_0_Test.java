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

public class Artists_toString_4_0_Test {

    @Test
    void testToString() {
        Artists artists = new Artists();
        artists.setArtist(new String[] { "artist1", "artist2" });
        String result = artists.toString();
        assertEquals(result, "# of Lists = 2\nartist - artist1\nartist - artist2\n");
    }
}
