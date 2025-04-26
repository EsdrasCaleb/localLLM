package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Platforms_toString_4_0_Test {

    private Platforms platforms;

    @BeforeEach
    public void setUp() {
        platforms = new Platforms();
    }

    @Test
    public void testToString() {
        String[] platformsArray = { "Android", "iOS", "Windows" };
        platforms.setPlatform(platformsArray);
        String output = platforms.toString();
        assertEquals("# of Platforms = 3\n", output);
        assertEquals("Platform - Android\n", output.substring(0, 10));
        assertEquals("Platform - iOS\n", output.substring(10, 20));
        assertEquals("Platform - Windows\n", output.substring(20, 30));
    }

    @Test
    public void testToStringEmptyPlatformList() {
        String[] emptyPlatformsArray = {};
        platforms.setPlatform(emptyPlatformsArray);
        String output = platforms.toString();
        assertEquals("# of Platforms = 0\n", output);
    }
}
