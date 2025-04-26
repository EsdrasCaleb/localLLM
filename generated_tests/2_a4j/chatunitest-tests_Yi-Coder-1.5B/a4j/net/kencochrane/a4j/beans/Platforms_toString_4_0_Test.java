package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class Platforms_toString_4_0_Test {

    @Test
    public void testToString() {
        Platforms platforms = new Platforms();
        platforms.setPlatform(new String[] { "Google", "Yahoo", "IBM" });
        String expected = "# of Platforms = 3\n" + "Platform - Google\n" + "Platform - Yahoo\n" + "Platform - IBM\n";
        assertEquals(expected, platforms.toString());
    }
}
