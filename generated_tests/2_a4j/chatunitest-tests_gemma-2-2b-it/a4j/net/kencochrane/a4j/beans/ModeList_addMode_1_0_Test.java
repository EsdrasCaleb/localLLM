package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class ModeList_addMode_1_0_Test {

    @Test
    void addMode_ShouldAddModeToArrayList() {
        // Arrange
        ModeList modeList = new ModeList();
        Mode mode = Mockito.mock(Mode.class);
        // Act
        modeList.addMode(mode);
        // Assert
        assertEquals(1, modeList.getAllModes().size());
        assertTrue(modeList.getAllModes().contains(mode));
    }
}
