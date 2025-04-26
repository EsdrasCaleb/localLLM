package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class Starring_getActor_3_4_Test {

    @Mock
    private Starring focal;

    @InjectMocks
    private Starring instance;

    @Test
    public void testGetActor() {
        // Arrange
        String[] actors = new String[] { "Actor1", "Actor2", "Actor3" };
        when(focal.getActor(0)).thenReturn(actors[0]);
        // Act
        String result = focal.getActor(0);
        // Assert
        assertEquals("Actor1", result);
    }
}
