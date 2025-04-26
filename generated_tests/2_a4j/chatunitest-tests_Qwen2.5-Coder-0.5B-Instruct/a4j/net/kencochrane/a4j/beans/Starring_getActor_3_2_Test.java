package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Starring_getActor_3_2_Test {

    @Test
    void testGetActor() {
        // Arrange
        Starring starring = new Starring();
        ArrayList<String> actors = new ArrayList<>();
        actors.add("Tom Hanks");
        actors.add("Brad Pitt");
        starring.setActor(actors.toArray(new String[0]));
        int index = 1;
        // Act
        String result = starring.getActor(index);
        // Assert
        assertEquals("Brad Pitt", result);
    }
}
