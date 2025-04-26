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

class Artists_getArtist_3_0_Test {

    @Test
    void getArtist_validIndex_returnsArtist() {
        // Arrange
        ArrayList<String> artistsList = new ArrayList<>(Arrays.asList("Artist1", "Artist2", "Artist3"));
        Artists artists = new Artists();
        artists.setArtist(artistsList.toArray(new String[0]));
        // Act
        String artist = artists.getArtist(1);
        // Assert
        assertEquals("Artist2", artist);
    }

    @Test
    void getArtist_invalidIndex_returnsNull() {
        // Arrange
        ArrayList<String> artistsList = new ArrayList<>(Arrays.asList("Artist1", "Artist2"));
        Artists artists = new Artists();
        artists.setArtist(artistsList.toArray(new String[0]));
        // Act
        String artist = artists.getArtist(2);
        // Assert
        assertNull(artist);
    }

    @Test
    void getArtist_emptyArrayList_returnsNull() {
        // Arrange
        Artists artists = new Artists();
        // Important: initialize with empty array
        artists.setArtist(new String[0]);
        // Act
        String artist = artists.getArtist(0);
        // Assert
        assertNull(artist);
    }

    @Test
    void getArtist_indexEqualsSizeMinusOne_returnsLastElement() {
        // Arrange
        ArrayList<String> artistsList = new ArrayList<>(Arrays.asList("Artist1", "Artist2"));
        Artists artists = new Artists();
        artists.setArtist(artistsList.toArray(new String[0]));
        // Act
        String artist = artists.getArtist(1);
        // Assert
        assertEquals("Artist2", artist);
    }

    @Test
    void getArtist_negativeIndex_returnsNull() {
        // Arrange
        ArrayList<String> artistsList = new ArrayList<>(Arrays.asList("Artist1", "Artist2"));
        Artists artists = new Artists();
        artists.setArtist(artistsList.toArray(new String[0]));
        // Act
        String artist = artists.getArtist(-1);
        // Assert
        assertNull(artist);
    }
}
