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

class Artists_getArtist_3_2_Test {

    @InjectMocks
    private Artists artists;

    @Mock
    private ArrayList<String> mockArtists;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        artists = new Artists();
        artists.setArtist(new String[] { "Artist1", "Artist2", "Artist3" });
    }

    @Test
    void testGetArtistValidIndex() {
        String result = artists.getArtist(1);
        assertEquals("Artist2", result);
    }

    @Test
    void testGetArtistInvalidIndex() {
        String result = artists.getArtist(5);
        assertNull(result);
    }

    @Test
    void testGetArtistEmptyList() {
        artists.setArtist(new String[] {});
        String result = artists.getArtist(0);
        assertNull(result);
    }

    @Test
    void testGetArtistBoundaryIndex() {
        String result = artists.getArtist(2);
        assertEquals("Artist3", result);
    }

    @Test
    void testGetArtistNegativeIndex() {
        String result = artists.getArtist(-1);
        assertNull(result);
    }
}
