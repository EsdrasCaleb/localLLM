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
public class Directors_getDirector_3_0_Test {

    @Mock
    private Directors directors;

    @InjectMocks
    private Directors director;

    @Test
    public void testGetDirector() {
        // Given
        String[] directorsArray = { "John", "Alice", "Bob" };
        director.setDirector(directorsArray);
        // When
        String expectedDirector = director.getDirector(0);
        // Then
        assertEquals("John", expectedDirector);
    }
}
