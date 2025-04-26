package net.kencochrane.a4j.beans;

import java.lang.reflect.Method;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Artists_getArtist_3_0_Test {

    private Artists artists;

    @BeforeEach
    void setUp() {
        artists = new Artists();
    }

    @Test
    void testGetArtist_NegativeIndex() throws Exception {
        String[] artistNames = { "Artist1", "Artist2", "Artist3" };
        artists.setArtist(artistNames);
        // Invoke the focal method
        Method method = Artists.class.getDeclaredMethod("getArtist", int.class);
        method.setAccessible(true);
        // Negative index
        String result = (String) method.invoke(artists, -1);
        assertNull(result);
    }
}
