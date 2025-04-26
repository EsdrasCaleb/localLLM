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

    @Test
    void testToString() {
        Platforms platforms = new Platforms();
        platforms.setPlatform(new String[] { "Android", "iOS" });
        String expectedOutput = "# of Platforms = 2\nPlatform - Android\nPlatform - iOS\n";
        String actualOutput = platforms.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
