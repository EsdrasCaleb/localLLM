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
    public void testToString_InvalidInput() {
        Platforms platforms = new Platforms();
        String[] invalidInput = null;
        assertThrows(NullPointerException.class, () -> platforms.setPlatform(invalidInput));
    }
}
