package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Platforms_getPlatform_3_1_Test {

    @Test
    public void testGetPlatform() {
        Platforms platforms = new Platforms();
        platforms.setPlatform(new String[] { "Android", "iOS", "Windows", "MacOS" });
        assertEquals("Android", platforms.getPlatform(0));
        assertEquals("iOS", platforms.getPlatform(1));
        assertEquals("Windows", platforms.getPlatform(2));
        assertEquals("MacOS", platforms.getPlatform(3));
        assertEquals(null, platforms.getPlatform(4));
    }
}
