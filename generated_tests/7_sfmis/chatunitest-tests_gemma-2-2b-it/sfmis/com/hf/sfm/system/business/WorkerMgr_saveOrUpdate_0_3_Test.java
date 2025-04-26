package com.hf.sfm.system.business;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.hibernate.Transaction;
import com.hf.sfm.sfmis.personinfo.business.PersonInfoMgr;
import com.hf.sfm.system.pdo.AWorker;
import com.hf.sfm.util.DaoFactory;

public class WorkerMgr_saveOrUpdate_0_3_Test {

    @Test
    void saveOrUpdate() {
        // Given
        WorkerMgr workerMgr = Mockito.mock(WorkerMgr.class);
        AWorker worker = new AWorker();
        worker.setIdno("123456789");
        worker.setPassword("1234");
        // When
        String result = workerMgr.saveOrUpdate(worker);
        // Then
        assertEquals("1", result);
    }
}
