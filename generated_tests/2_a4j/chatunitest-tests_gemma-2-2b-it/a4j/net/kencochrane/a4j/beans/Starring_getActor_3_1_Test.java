package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Starring_getActor_3_1_Test {

    @Test
    void testGetActor() {
        Starring starring = mock(Starring.class);
        when(starring.getActor(0)).thenReturn("John");
        String actor = starring.getActor(0);
        assertEquals("John", actor);
    }
}
