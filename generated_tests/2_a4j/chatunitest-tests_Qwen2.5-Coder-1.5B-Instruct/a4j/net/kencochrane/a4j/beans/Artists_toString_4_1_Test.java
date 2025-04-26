package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Artists_toString_4_1_Test {

    @Test
    public void testToStringWithEmptyList() {
        Artists artists = new Artists();
        assertEquals("artists is null or size 0 \n", artists.toString());
    }

    @Test
    public void testToStringWithSingleElementList() {
        Artists artists = new Artists();
        artists.setArtist(new String[] { "John Doe" });
        assertEquals("# of Lists = 1\nartist - John Doe\n", artists.toString());
    }

    @Test
    public void testToStringWithMultipleElementsList() {
        Artists artists = new Artists();
        artists.setArtist(new String[] { "John Doe", "Jane Smith", "Alice Johnson" });
        assertEquals("# of Lists = 3\nartist - John Doe\nartist - Jane Smith\nartist - Alice Johnson\n", artists.toString());
    }
}
