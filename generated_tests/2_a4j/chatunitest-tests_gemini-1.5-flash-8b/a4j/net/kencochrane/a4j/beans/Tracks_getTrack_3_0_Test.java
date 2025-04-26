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

class Tracks_getTrack_3_0_Test {

    @Test
    void getTrack_negativeIndex() {
        // Arrange
        Tracks tracks = new Tracks();
        String[] tracksArray = { "Track 1", "Track 2", "Track 3" };
        tracks.setTrack(tracksArray);
        // Act
        String track = tracks.getTrack(-1);
        // Assert
        assertNull(track);
    }
}
