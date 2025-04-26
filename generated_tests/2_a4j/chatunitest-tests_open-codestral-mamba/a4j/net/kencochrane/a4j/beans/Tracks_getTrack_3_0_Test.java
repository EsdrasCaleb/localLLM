package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class Tracks_getTrack_3_0_Test {

    @Mock
    private ArrayList tracks;

    private Tracks tracksUnderTest;

    @BeforeEach
    public void setUp() throws Exception {
        tracksUnderTest = new Tracks();
        Field tracksField = Tracks.class.getDeclaredField("tracks");
        tracksField.setAccessible(true);
        tracksField.set(tracksUnderTest, tracks);
    }

    @Test
    public void testGetTrackWithinRange() {
        when(tracks.size()).thenReturn(3);
        when(tracks.get(0)).thenReturn("Track1");
        when(tracks.get(1)).thenReturn("Track2");
        when(tracks.get(2)).thenReturn("Track3");
        assertEquals("Track2", tracksUnderTest.getTrack(1));
    }

    @Test
    public void testGetTrackOutsideRange() {
        when(tracks.size()).thenReturn(3);
        assertNull(tracksUnderTest.getTrack(3));
    }

    @Test
    public void testGetTrackEmptyList() {
        when(tracks.size()).thenReturn(0);
        assertNull(tracksUnderTest.getTrack(0));
    }
}
