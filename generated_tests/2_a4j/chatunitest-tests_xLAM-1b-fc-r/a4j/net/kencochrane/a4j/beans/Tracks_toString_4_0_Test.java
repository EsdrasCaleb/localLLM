package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Tracks_toString_4_0_Test {

    @Test
    void testToString() {
        Tracks tracks = new Tracks();
        tracks.setTrack(new String[] { "Track 1", "Track 2", "Track 3" });
        String expectedOutput = "# of Tracks = 3\n" + "Track - Track 1\n" + "Track - Track 2\n" + "Track - Track 3\n";
        assertEquals(expectedOutput, tracks.toString());
    }
}
