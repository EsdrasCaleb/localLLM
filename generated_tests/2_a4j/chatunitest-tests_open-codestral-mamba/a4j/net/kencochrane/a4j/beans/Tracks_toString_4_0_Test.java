package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Tracks_toString_4_0_Test {

    @Mock
    private ArrayList<String> tracks;

    @InjectMocks
    private Tracks trackUnderTest;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testToString() {
        when(tracks.size()).thenReturn(3);
        when(tracks.get(0)).thenReturn("Track1");
        when(tracks.get(1)).thenReturn("Track2");
        when(tracks.get(2)).thenReturn("Track3");
        String expectedOutput = "# of Tracks = 3\n" + "Track - Track1\n" + "Track - Track2\n" + "Track - Track3\n";
        String actualOutput = trackUnderTest.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
