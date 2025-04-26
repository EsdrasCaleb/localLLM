package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Platforms_toString_4_0_Test {

    @Test
    public void testToString() {
        Platforms platforms = new Platforms();
        platforms.setPlatform(new String[] { "Platform 1", "Platform 2", "Platform 3" });
        String expectedOutput = "# of Platforms = 3\n" + "Platform - Platform 1\n" + "Platform - Platform 2\n" + "Platform - Platform 3\n";
        assertEquals(expectedOutput, platforms.toString());
    }
}
