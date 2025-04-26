package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Tracks_getTrack_3_0_Test {

    @Test
    public void testGetTrack_NullArray_ThrowsNullPointerException() {
        Tracks tracks = new Tracks();
        String[] nullArray = null;
        assertThrows(NullPointerException.class, () -> tracks.getTrack(0));
    }

    @Test
    public void testGetTrack_InvalidIndex_ThrowsIndexOutOfBoundsException() {
        Tracks tracks = new Tracks();
        tracks.setTrack(new String[] { "track1", "track2", "track3" });
        assertThrows(IndexOutOfBoundsException.class, () -> tracks.getTrack(3));
    }
}
