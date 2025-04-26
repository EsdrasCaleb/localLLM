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

class Platforms_toString_4_0_Test {

    private Platforms platforms;

    @BeforeEach
    void setUp() {
        platforms = new Platforms();
    }

    @Test
    void testToString_emptyPlatform() {
        assertEquals("Platforms is null or size 0\n", platforms.toString());
    }

    @Test
    void testToString_singlePlatform() {
        String[] platformsArray = { "Linux" };
        platforms.setPlatform(platformsArray);
        assertEquals("# of Platforms = 1\nPlatform - Linux\n", platforms.toString());
    }

    @Test
    void testToString_multiplePlatforms() {
        String[] platformsArray = { "Linux", "Windows", "MacOS" };
        platforms.setPlatform(platformsArray);
        assertEquals("# of Platforms = 3\nPlatform - Linux\nPlatform - Windows\nPlatform - MacOS\n", platforms.toString());
    }
}
