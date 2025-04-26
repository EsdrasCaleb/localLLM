package com.hf.sfm.system.business;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.hibernate.Transaction;
import com.hf.sfm.sfmis.personinfo.business.PersonInfoMgr;
import com.hf.sfm.system.pdo.AWorker;
import com.hf.sfm.util.DaoFactory;

@ExtendWith(MockitoExtension.class)
public class WorkerMgr_saveOrUpdate_0_0_Test {

    @Mock
    private WorkerMgr focalClass;

    @InjectMocks
    private AWorker worker;

    @Test
    public void testSaveOrUpdateNewWorker() {
        // Arrange
        when(focalClass.encrypt(worker.getPassword())).thenReturn("encryptedPassword");
        doNothing().when(focalClass).save(worker);
        // Act
        String result = focalClass.saveOrUpdate(worker);
        // Assert
        assertEquals("1", result);
    }

    @Test
    public void testSaveOrUpdateExistingWorker() {
        // Arrange
        when(focalClass.encrypt(worker.getPassword())).thenReturn("encryptedPassword");
        doNothing().when(focalClass).update(worker);
        // Act
        String result = focalClass.saveOrUpdate(worker);
        // Assert
        assertEquals("1", result);
    }

    @Test
    public void testSaveOrUpdateException() {
        // Arrange
        when(focalClass.encrypt(worker.getPassword())).thenThrow(new RuntimeException("Encryption error"));
        // Act
        String result = focalClass.saveOrUpdate(worker);
        // Assert
        assertEquals("0", result);
    }
}
