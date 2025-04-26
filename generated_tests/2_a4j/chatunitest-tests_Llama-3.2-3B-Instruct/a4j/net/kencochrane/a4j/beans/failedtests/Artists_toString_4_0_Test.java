package net.kencochrane.a4j.beans;

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
    public void testToString_EmptyList() {
        Artists artists = new Artists();
        String result = artists.toString();
        assertNotNull(result);
        assertEquals("", result);
    }

    @Test
    public void testToString_SingleElementList() {
        Artists artists = new Artists();
        artists.setArtist(new String[] { "John" });
        String result = artists.toString();
        assertNotNull(result);
        assertEquals("artist - John\n", result);
    }

    @Test
    public void testToString_MultipleElementList() {
        Artists artists = new Artists();
        artists.setArtist(new String[] { "John", "Jane", "Bob" });
        String result = artists.toString();
        assertNotNull(result);
        assertEquals("artist - John\nartist - Jane\nartist - Bob\n", result);
    }

    @Test
    public void testToString_NullList() {
        Artists artists = new Artists();
        artists.setArtist(null);
        String result = artists.toString();
        assertNotNull(result);
        assertEquals("artists is null or size 0 \n", result);
    }

    @Test
    public void testToString_NullArtist() {
        Artists artists = new Artists();
        String result = artists.toString();
        assertNotNull(result);
        assertEquals("artists is null or size 0 \n", result);
    }
}
