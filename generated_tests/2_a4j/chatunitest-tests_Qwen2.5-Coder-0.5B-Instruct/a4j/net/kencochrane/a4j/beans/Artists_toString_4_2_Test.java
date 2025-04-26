// Test class
package net.kencochrane.a4j.beans;

import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;

class Artists_toString_4_2_Test {

    // <Buggy Line>: incompatible types: org.mockito.MockitoAnnotations cannot be converted to java.lang.annotation.Annotation
    @InjectMocks
    private Artists artistsMock;

    @Test
    public void testToString() {
        // Arrange
        MockitoAnnotations.initMocks(this);
        // Act
        Artists artists = new Artists();
        artists.setArtist(new String[] { "Artist1", "Artist2" });
        artistsMock.setArtist(artists.getArtist());
        // Assert
        assertEquals("artists is null or size 0 \nartist - Artist1\nartist - Artist2", artistsMock.toString());
    }
}
