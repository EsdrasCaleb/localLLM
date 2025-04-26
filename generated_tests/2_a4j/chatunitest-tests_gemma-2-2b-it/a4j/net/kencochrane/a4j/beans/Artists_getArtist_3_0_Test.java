package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Artists_getArtist_3_0_Test {

    @Test
    void testGetArtist() {
        Artists artists = new Artists();
        artists.setArtist(new String[] { "John", "Jane" });
        String expected = "John";
        assertEquals(expected, artists.getArtist(0));
    }
}
