package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Artists_getArtist_3_0_Test {

    private Artists artists;

    @BeforeEach
    public void setUp() {
        artists = new Artists();
        artists.artists = new ArrayList<>();
        artists.artists.add("Artist1");
        artists.artists.add("Artist2");
        artists.artists.add("Artist3");
    }

    @Test
    public void testGetArtist() throws Exception {
        // Arrange
        int index = 1;
        String expected = "Artist2";
        // Act
        String result = artists.getArtist(index);
        // Assert
        org.junit.jupiter.api.Assertions.assertEquals(expected, result);
    }
}
