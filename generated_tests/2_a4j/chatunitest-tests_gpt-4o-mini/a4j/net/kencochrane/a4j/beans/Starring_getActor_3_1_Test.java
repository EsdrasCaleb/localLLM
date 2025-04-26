package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Starring_getActor_3_1_Test {

    private Starring starring;

    @BeforeEach
    public void setUp() {
        starring = new Starring();
    }

    @Test
    public void testGetActor_NegativeIndex() throws Exception {
        // Arrange
        String[] actorsArray = { "Actor 1", "Actor 2", "Actor 3" };
        setActorsField(starring, actorsArray);
        // Act
        // negative index
        String result = starring.getActor(-1);
        // Assert
        assertNull(result);
    }

    private void setActorsField(Starring starring, String[] actorsArray) throws Exception {
        Field actorsField = Starring.class.getDeclaredField("actors");
        actorsField.setAccessible(true);
        ArrayList<String> actorsList = new ArrayList<>();
        for (String actor : actorsArray) {
            actorsList.add(actor);
        }
        actorsField.set(starring, actorsList);
    }
}
