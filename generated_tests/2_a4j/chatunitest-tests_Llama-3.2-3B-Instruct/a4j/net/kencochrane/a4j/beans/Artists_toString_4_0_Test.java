package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Artists_toString_4_0_Test {

    @Test
    public void testToString_NullArtist() {
        Artists artists = new Artists();
        String result = artists.toString();
        assertNotNull(result);
        assertEquals("artists is null or size 0 \n", result);
    }
}
