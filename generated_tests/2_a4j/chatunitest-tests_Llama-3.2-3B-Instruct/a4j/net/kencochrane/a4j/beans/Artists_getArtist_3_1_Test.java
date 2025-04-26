package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Artists_getArtist_3_1_Test {

    @Test
    public void testGetArtist_InValidIndex_ReturnsNull() {
        Artists artists = new Artists();
        artists.artists = new ArrayList<>();
        assertEquals(null, artists.getArtist(-1));
    }

    @Test
    public void testGetArtist_NullArtistsList_ThrowsNullPointerException() {
        Artists artists = new Artists();
        assertThrows(NullPointerException.class, () -> artists.getArtist(0));
    }
}
