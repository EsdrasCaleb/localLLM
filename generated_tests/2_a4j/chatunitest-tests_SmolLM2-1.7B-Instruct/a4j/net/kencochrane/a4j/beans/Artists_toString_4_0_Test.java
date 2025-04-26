package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class Artists_toString_4_0_Test {

    @Test
    public void testToString() {
        Artists artists = new Artists();
        artists.setArtist(new String[] { "John", "Paul", "George", "Ringo" });
        assertEquals("# of Lists = 4\nartist - John\nartist - Paul\nartist - George\nartist - Ringo", artists.toString());
        artists.setArtist(null);
        assertEquals("artists is null or size 0 \n", artists.toString());
        artists.setArtist(new String[0]);
        assertEquals("artists is null or size 0 \n", artists.toString());
    }
}
