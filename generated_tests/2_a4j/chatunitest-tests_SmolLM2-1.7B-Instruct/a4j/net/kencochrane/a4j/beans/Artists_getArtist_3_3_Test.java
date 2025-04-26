package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
class Artists_getArtist_3_3_Test {

    @Mock
    private Artists artists;

    @InjectMocks
    private Artists focal;

    @Test
    void testGetArtist() {
        // Arrange
        artists.setArtist(new String[] { "Artist1", "Artist2" });
        artists.setArtist(new String[] { "Artist3", "Artist4" });
        // Act
        focal.setArtist(new String[] { "Artist5", "Artist6" });
        // Assert
        assert focal.getArtist(0).equals("Artist1");
        assert focal.getArtist(1).equals("Artist3");
    }
}
