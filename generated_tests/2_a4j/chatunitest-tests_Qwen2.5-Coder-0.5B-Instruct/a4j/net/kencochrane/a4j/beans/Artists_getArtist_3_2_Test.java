package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Artists_getArtist_3_2_Test {

    @Test
    public void testGetArtist() {
        // Arrange
        MockitoAnnotations.openMocks(Artists.class);
        Artists artists = Mockito.mock(Artists.class);
        artists.setArtist(new String[] { "Artist1", "Artist2", "Artist3" });
        int index = 1;
        // Act
        String artist = artists.getArtist(index);
        // Assert
        assertEquals("Artist2", artist);
    }
}
