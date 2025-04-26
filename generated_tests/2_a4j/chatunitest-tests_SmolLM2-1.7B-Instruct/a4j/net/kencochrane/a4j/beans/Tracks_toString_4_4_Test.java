package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
class Tracks_toString_4_4_Test {

    @Mock
    private Tracks mockTracks;

    @InjectMocks
    private Tracks tracks;

    @Test
    void testToString() {
        // Arrange
        ArrayList<String> tracks = new ArrayList<>(Arrays.asList("Track1", "Track2"));
        tracks.add("Track3");
        tracks.add(null);
        // Act
        String output = tracks.toString();
        // Assert
        assert output.contains("# of Tracks = 3");
        assert output.contains("Track - Track1");
        assert output.contains("Track - Track2");
        assert output.contains("Track - Track3");
        assert output.contains("Tracks is null or size 0");
    }
}
