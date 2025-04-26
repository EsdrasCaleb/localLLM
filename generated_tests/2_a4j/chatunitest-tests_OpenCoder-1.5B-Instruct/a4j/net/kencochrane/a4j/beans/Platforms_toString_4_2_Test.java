package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Platforms_toString_4_2_Test {

    Platforms platforms;

    @Test
    public void testToString() {
        platforms = new Platforms();
        platforms.setPlatform(new String[] { "PS4", "Xbox One" });
        assertEquals("Platforms is null or size 0\n# of Platforms = 2\nPlatform - PS4\nPlatform - Xbox One\n", platforms.toString());
    }
}
