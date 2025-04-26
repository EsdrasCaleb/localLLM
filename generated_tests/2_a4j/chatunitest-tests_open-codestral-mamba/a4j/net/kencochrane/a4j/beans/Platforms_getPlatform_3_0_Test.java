package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Platforms_getPlatform_3_0_Test {

    @Mock
    private Platforms platforms;

    @Test
    public void testGetPlatformWithinBounds() {
        ArrayList<String> platformList = new ArrayList<>(Arrays.asList("Platform1", "Platform2", "Platform3"));
        when(platforms.getPlatform(1)).thenReturn("Platform2");
        assertEquals("Platform2", platforms.getPlatform(1));
    }

    @Test
    public void testGetPlatformOutOfBounds() {
        when(platforms.getPlatform(3)).thenReturn(null);
        assertNull(platforms.getPlatform(3));
    }

    @Test
    public void testGetPlatformEmptyList() {
        when(platforms.getPlatform(0)).thenReturn(null);
        assertNull(platforms.getPlatform(0));
    }
}
